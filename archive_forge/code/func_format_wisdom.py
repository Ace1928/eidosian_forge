from reportlab import rl_config
def format_wisdom(text, line_length=72):
    try:
        import textwrap
        return textwrap.fill(text, line_length)
    except:
        return text