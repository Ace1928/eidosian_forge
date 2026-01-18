from parlai.core.torch_agent import TorchAgent
from .controls import eval_attr
def fix_personaline_period(line):
    """
    Sometimes the tokenized persona lines have a period at the end but no space before
    the period.

    This function fixes it, e.g. changes 'my favorite color is blue.' to 'my favorite
    color is blue .'
    """
    assert len(line) >= 2
    assert line[-1] == '.' and line[-2] != ' '
    pl = line[:-1] + ' .'
    return pl