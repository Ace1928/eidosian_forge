import math
import warnings
import matplotlib.dates
def get_spine_visible(ax, spine_key):
    """Return some spine parameters for the spine, `spine_key`."""
    spine = ax.spines[spine_key]
    ax_frame_on = ax.get_frame_on()
    position = spine._position or ('outward', 0.0)
    if isinstance(position, str):
        if position == 'center':
            position = ('axes', 0.5)
        elif position == 'zero':
            position = ('data', 0)
    position_type, amount = position
    if position_type == 'outward' and amount == 0:
        spine_frame_like = True
    else:
        spine_frame_like = False
    if not spine.get_visible():
        return False
    elif not spine._edgecolor[-1]:
        return False
    elif not ax_frame_on and spine_frame_like:
        return False
    elif ax_frame_on and spine_frame_like:
        return True
    elif not ax_frame_on and (not spine_frame_like):
        return True
    else:
        return False