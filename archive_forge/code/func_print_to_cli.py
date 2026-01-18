from __future__ import annotations
def print_to_cli(message: str, **kwargs) -> None:
    """Print a message to the terminal using click if available, else print
    using the built-in print function.

    You can provide any keyword arguments that click.secho supports.
    """
    try:
        import click
        click.secho(message, **kwargs)
    except ImportError:
        print(message, flush=True)