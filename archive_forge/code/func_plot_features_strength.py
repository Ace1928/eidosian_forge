import math
from . import _catboost
from .core import CatBoost, CatBoostError
from .utils import _import_matplotlib
def plot_features_strength(model, height_per_feature=0.5, width_per_plot=5, plots_per_row=None):
    with _import_matplotlib() as _plt:
        plt = _plt
    strengths = calc_features_strength(model)
    dimension = len(strengths[0])
    features = len(strengths)
    if not plots_per_row:
        plots_per_row = min(5, dimension)
    rows = int(math.ceil(dimension / plots_per_row))
    fig, axes = plt.subplots(rows, plots_per_row)
    if rows == 1:
        axes = [axes]
    if plots_per_row == 1:
        axes = [[row_axes] for row_axes in axes]
    fig.suptitle('Features Strength')
    fig.set_size_inches(width_per_plot * plots_per_row, height_per_feature * features * rows)
    for dim in range(dimension):
        strengths = [(s[dim], i) for i, s in enumerate(strengths)]
        strengths = list(sorted(strengths))
        labels = ['Feature #{}'.format(f) for _, f in strengths]
        strengths = [s for s, _ in strengths]
        ax = axes[dim // plots_per_row][dim % plots_per_row]
        colors = [(1, 0, 0) if s > 0 else (0, 0, 1) for s in strengths]
        ax.set_title('Dimension={}'.format(dim))
        ax.barh(range(len(strengths)), strengths, align='center', color=colors)
        ax.set_yticks(range(len(strengths)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Prediction value change')
    return fig