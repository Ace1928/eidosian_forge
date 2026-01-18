from parlai.core.torch_agent import TorchAgent
from .controls import eval_attr
def remove_prefix(utt, prefix):
    """
    Check that utt begins with prefix+" ", and then remove.

    Inputs:
      utt: string
      prefix: string

    Returns:
      new utt: utt with the prefix+" " removed.
    """
    try:
        assert utt[:len(prefix) + 1] == prefix + ' '
    except AssertionError as e:
        print("ERROR: utterance '%s' does not start with '%s '" % (utt, prefix))
        print(repr(utt[:len(prefix) + 1]))
        print(repr(prefix + ' '))
        raise e
    return utt[len(prefix) + 1:]