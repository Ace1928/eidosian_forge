import binascii
import codecs
import marshal
import os
import types as python_types
def remove_long_seq(maxlen, seq, label):
    """Removes sequences that exceed the maximum length.

    Args:
        maxlen: Int, maximum length of the output sequences.
        seq: List of lists, where each sublist is a sequence.
        label: List where each element is an integer.

    Returns:
        new_seq, new_label: shortened lists for `seq` and `label`.
    """
    new_seq, new_label = ([], [])
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return (new_seq, new_label)