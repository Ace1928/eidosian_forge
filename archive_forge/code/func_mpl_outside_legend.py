def mpl_outside_legend(ax, **kwargs):
    """ Places a legend box outside a matplotlib Axes instance. """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), **kwargs)