from fontTools.pens.basePen import AbstractPen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, DecomposingPointPen
def replayRecording(recording, pen):
    """Replay a recording, as produced by RecordingPen or DecomposingRecordingPen,
    to a pen.

    Note that recording does not have to be produced by those pens.
    It can be any iterable of tuples of method name and tuple-of-arguments.
    Likewise, pen can be any objects receiving those method calls.
    """
    for operator, operands in recording:
        getattr(pen, operator)(*operands)