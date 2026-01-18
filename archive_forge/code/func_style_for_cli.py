from __future__ import annotations
def style_for_cli(message: str, **kwargs) -> str:
    """Style a message using click if available, else return the message
    unchanged.

    You can provide any keyword arguments that click.style supports.
    """
    try:
        import click
        return click.style(message, **kwargs)
    except ImportError:
        return message