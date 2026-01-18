from reportlab.rl_config import register_reset
def getCodeNames():
    """Returns sorted list of supported bar code names"""
    return sorted(getCodes().keys())