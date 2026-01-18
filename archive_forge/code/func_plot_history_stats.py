import itertools
import functools
import importlib.util
@default_to_neutral_style
def plot_history_stats(self, *, fn='count', colors=None, rasterize_dpi=300, ax=None, figsize=(2, 2), show_and_close=True):
    from matplotlib import pyplot as plt
    stats = self.history_stats(fn)
    colors = get_default_colors_dict(colors)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_dpi(rasterize_dpi)
    else:
        fig = None
    xs, labels, clrs = ([], [], [])
    for fn_name, cnt in sorted(stats.items(), key=lambda x: -x[1]):
        xs.append(cnt)
        labels.append(f'{fn_name}: {cnt}')
        try:
            color = colors[fn_name]
        except KeyError:
            color = colors[fn_name] = hash_to_color(fn_name)
        clrs.append(color)
    ax.pie(x=xs, labels=labels, colors=clrs)
    if fig is not None and show_and_close:
        plt.show()
        plt.close(fig)
    return (fig, ax)