from collections import defaultdict
import click
def cb_layer(ctx, param, value):
    """Let --layer be a name or index."""
    if value is None or not value.isdigit():
        return value
    else:
        return int(value)