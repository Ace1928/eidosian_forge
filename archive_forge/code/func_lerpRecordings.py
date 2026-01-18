from fontTools.pens.basePen import AbstractPen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, DecomposingPointPen
def lerpRecordings(recording1, recording2, factor=0.5):
    """Linearly interpolate between two recordings. The recordings
    must be decomposed, i.e. they must not contain any components.

    Factor is typically between 0 and 1. 0 means the first recording,
    1 means the second recording, and 0.5 means the average of the
    two recordings. Other values are possible, and can be useful to
    extrapolate. Defaults to 0.5.

    Returns a generator with the new recording.
    """
    if len(recording1) != len(recording2):
        raise ValueError('Mismatched lengths: %d and %d' % (len(recording1), len(recording2)))
    for (op1, args1), (op2, args2) in zip(recording1, recording2):
        if op1 != op2:
            raise ValueError('Mismatched operations: %s, %s' % (op1, op2))
        if op1 == 'addComponent':
            raise ValueError('Cannot interpolate components')
        else:
            mid_args = [(x1 + (x2 - x1) * factor, y1 + (y2 - y1) * factor) for (x1, y1), (x2, y2) in zip(args1, args2)]
        yield (op1, mid_args)