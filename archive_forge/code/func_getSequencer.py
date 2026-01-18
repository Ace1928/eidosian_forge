from reportlab.rl_config import register_reset
def getSequencer():
    global _sequencer
    if _sequencer is None:
        _sequencer = Sequencer()
    return _sequencer