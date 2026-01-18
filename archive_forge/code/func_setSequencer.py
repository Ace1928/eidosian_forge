from reportlab.rl_config import register_reset
def setSequencer(seq):
    global _sequencer
    s = _sequencer
    _sequencer = seq
    return s