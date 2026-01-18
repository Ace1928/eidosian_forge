from collections import Counter
from . import EvaluationMixin, evaluation_io
from ..io import load_key
def key_label_to_class(key_label):
    """
    Convert key label to key class number.

    The key label must follow the MIREX syntax defined at
    http://music-ir.org/mirex/wiki/2017:Audio_Key_Detection:
    `tonic mode`, where tonic is in {C, C#, Db, ... Cb} and mode in {'major',
    'maj', 'minor', 'min'}. The label will be converted into a class id based
    on the root pitch id (c .. 0, c# .. 1, ..., cb ... 11) plus 12 if in minor
    mode.

    Parameters
    ----------
    key_label : str
        Key label.

    Returns
    -------
    key_class : int
        Key class.

    Examples
    --------
    >>> from madmom.evaluation.key import key_label_to_class
    >>> key_label_to_class('D major')
    2

    >>> key_label_to_class('D minor')
    14

    """
    tonic, mode = key_label.split()
    if tonic.lower() not in _KEY_TO_SEMITONE.keys():
        raise ValueError('Unknown tonic: {}'.format(tonic))
    key_class = _KEY_TO_SEMITONE[tonic.lower()]
    if mode in ['minor', 'min']:
        key_class += 12
    elif mode in ['major', 'maj']:
        key_class += 0
    else:
        raise ValueError('Unknown mode: {}'.format(mode))
    return key_class