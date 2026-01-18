from reportlab.rl_config import register_reset
def getCodes():
    """Returns a dict mapping code names to widgets"""
    codes = {}
    for widget in _widgets:
        codeName = widget.codeName
        codes[codeName] = widget
    return codes