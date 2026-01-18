import tree
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell
from keras.src.layers.rnn.stacked_rnn_cells import StackedRNNCells
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
def inner_loop(self, sequences, initial_state, mask, training=False):
    cell_kwargs = {}
    if isinstance(self.cell, Layer) and self.cell._call_has_training_arg:
        cell_kwargs['training'] = training

    def step(inputs, states):
        output, new_states = self.cell(inputs, states, **cell_kwargs)
        if not tree.is_nested(new_states):
            new_states = [new_states]
        return (output, new_states)
    if not tree.is_nested(initial_state):
        initial_state = [initial_state]
    return backend.rnn(step, sequences, initial_state, go_backwards=self.go_backwards, mask=mask, unroll=self.unroll, input_length=sequences.shape[1], zero_output_for_mask=self.zero_output_for_mask, return_all_outputs=self.return_sequences)