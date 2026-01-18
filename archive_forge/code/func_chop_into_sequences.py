import logging
import numpy as np
import tree  # pip install dm_tree
from typing import List, Optional
import functools
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import TensorType, ViewRequirementsDict
from ray.util import log_once
from ray.rllib.utils.typing import SampleBatchType
@DeveloperAPI
def chop_into_sequences(*, feature_columns, state_columns, max_seq_len, episode_ids=None, unroll_ids=None, agent_indices=None, dynamic_max=True, shuffle=False, seq_lens=None, states_already_reduced_to_init=False, handle_nested_data=False, _extra_padding=0, padding: str='zero'):
    """Truncate and pad experiences into fixed-length sequences.

    Args:
        feature_columns: List of arrays containing features.
        state_columns: List of arrays containing LSTM state values.
        max_seq_len: Max length of sequences. Sequences longer than max_seq_len
            will be split into subsequences that span the batch dimension
            and sum to max_seq_len.
        episode_ids (List[EpisodeID]): List of episode ids for each step.
        unroll_ids (List[UnrollID]): List of identifiers for the sample batch.
            This is used to make sure sequences are cut between sample batches.
        agent_indices (List[AgentID]): List of agent ids for each step. Note
            that this has to be combined with episode_ids for uniqueness.
        dynamic_max: Whether to dynamically shrink the max seq len.
            For example, if max len is 20 and the actual max seq len in the
            data is 7, it will be shrunk to 7.
        shuffle: Whether to shuffle the sequence outputs.
        handle_nested_data: If True, assume that the data in
            `feature_columns` could be nested structures (of data).
            If False, assumes that all items in `feature_columns` are
            only np.ndarrays (no nested structured of np.ndarrays).
        _extra_padding: Add extra padding to the end of sequences.
        padding: Padding type to use. Either "zero" or "last". Zero padding
            will pad with zeros, last padding will pad with the last value.

    Returns:
        f_pad: Padded feature columns. These will be of shape
            [NUM_SEQUENCES * MAX_SEQ_LEN, ...].
        s_init: Initial states for each sequence, of shape
            [NUM_SEQUENCES, ...].
        seq_lens: List of sequence lengths, of shape [NUM_SEQUENCES].

    .. testcode::
        :skipif: True

        from ray.rllib.policy.rnn_sequencing import chop_into_sequences
        f_pad, s_init, seq_lens = chop_into_sequences(
            episode_ids=[1, 1, 5, 5, 5, 5],
            unroll_ids=[4, 4, 4, 4, 4, 4],
            agent_indices=[0, 0, 0, 0, 0, 0],
            feature_columns=[[4, 4, 8, 8, 8, 8],
                             [1, 1, 0, 1, 1, 0]],
            state_columns=[[4, 5, 4, 5, 5, 5]],
            max_seq_len=3)
        print(f_pad)
        print(s_init)
        print(seq_lens)


    .. testoutput::

        [[4, 4, 0, 8, 8, 8, 8, 0, 0],
         [1, 1, 0, 0, 1, 1, 0, 0, 0]]
        [[4, 4, 5]]
        [2, 3, 1]
    """
    if seq_lens is None or len(seq_lens) == 0:
        prev_id = None
        seq_lens = []
        seq_len = 0
        unique_ids = np.add(np.add(episode_ids, agent_indices), np.array(unroll_ids, dtype=np.int64) << 32)
        for uid in unique_ids:
            if prev_id is not None and uid != prev_id or seq_len >= max_seq_len:
                seq_lens.append(seq_len)
                seq_len = 0
            seq_len += 1
            prev_id = uid
        if seq_len:
            seq_lens.append(seq_len)
        seq_lens = np.array(seq_lens, dtype=np.int32)
    if dynamic_max:
        max_seq_len = max(seq_lens) + _extra_padding
    feature_sequences = []
    for col in feature_columns:
        if isinstance(col, list):
            col = np.array(col)
        feature_sequences.append([])
        for f in tree.flatten(col):
            if not isinstance(f, np.ndarray):
                f = np.array(f)
            length = len(seq_lens) * max_seq_len
            if f.dtype == object or f.dtype.type is np.str_:
                f_pad = [None] * length
            else:
                f_pad = np.zeros((length,) + np.shape(f)[1:], dtype=f.dtype)
            seq_base = 0
            i = 0
            for len_ in seq_lens:
                for seq_offset in range(len_):
                    f_pad[seq_base + seq_offset] = f[i]
                    i += 1
                if padding == 'last':
                    for seq_offset in range(len_, max_seq_len):
                        f_pad[seq_base + seq_offset] = f[i - 1]
                seq_base += max_seq_len
            assert i == len(f), f
            feature_sequences[-1].append(f_pad)
    if states_already_reduced_to_init:
        initial_states = state_columns
    else:
        initial_states = []
        for state_column in state_columns:
            if isinstance(state_column, list):
                state_column = np.array(state_column)
            initial_state_flat = []
            for s in tree.flatten(state_column):
                if not isinstance(s, np.ndarray):
                    s = np.array(s)
                s_init = []
                i = 0
                for len_ in seq_lens:
                    s_init.append(s[i])
                    i += len_
                initial_state_flat.append(np.array(s_init))
            initial_states.append(tree.unflatten_as(state_column, initial_state_flat))
    if shuffle:
        permutation = np.random.permutation(len(seq_lens))
        for i, f in enumerate(tree.flatten(feature_sequences)):
            orig_shape = f.shape
            f = np.reshape(f, (len(seq_lens), -1) + f.shape[1:])
            f = f[permutation]
            f = np.reshape(f, orig_shape)
            feature_sequences[i] = f
        for i, s in enumerate(initial_states):
            s = s[permutation]
            initial_states[i] = s
        seq_lens = seq_lens[permutation]
    if not handle_nested_data:
        feature_sequences = [f[0] for f in feature_sequences]
    return (feature_sequences, initial_states, seq_lens)