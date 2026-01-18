import math
from . import _catboost
from .core import CatBoost, CatBoostError
from .utils import _import_matplotlib
def plot_pdp(arg, size_per_plot=(5, 5), plots_per_row=None):
    with _import_matplotlib() as _plt:
        plt = _plt
    if isinstance(arg, CatBoost):
        arg = explain_features(arg)
    if isinstance(arg, _catboost.FeatureExplanation):
        arg = [arg]
    assert len(arg) > 0
    assert isinstance(arg, list)
    for element in arg:
        assert isinstance(element, _catboost.FeatureExplanation)
    figs = []
    for feature_explanation in arg:
        dimension = feature_explanation.dimension()
        if not plots_per_row:
            plots_per_row = min(5, dimension)
        rows = int(math.ceil(dimension / plots_per_row))
        fig, axes = plt.subplots(rows, plots_per_row)
        fig.suptitle('Feature #{}'.format(feature_explanation.feature))
        if rows == 1:
            axes = [axes]
        if plots_per_row == 1:
            axes = [[row_axes] for row_axes in axes]
        fig.set_size_inches(size_per_plot[0] * plots_per_row, size_per_plot[1] * rows)
        for dim in range(dimension):
            ax = axes[dim // plots_per_row][dim % plots_per_row]
            ax.set_title('Dimension={}'.format(dim))
            ax.set_xlabel('feature value')
            ax.set_ylabel('model value')
            borders, values = feature_explanation.calc_pdp(dim)
            xs = []
            ys = []
            if feature_explanation.type == 'Float':
                if len(borders) == 0:
                    xs.append(-0.1)
                    xs.append(0.1)
                    ys.append(feature_explanation.expected_bias[dim])
                    ys.append(feature_explanation.expected_bias[dim])
                    ax.plot(xs, ys)
                else:
                    offset = max(0.1, (borders[0] + borders[-1]) / 2)
                    xs.append(borders[0] - offset)
                    ys.append(feature_explanation.expected_bias[dim])
                    for border, value in zip(borders, values):
                        xs.append(border)
                        ys.append(ys[-1])
                        xs.append(border)
                        ys.append(value)
                    xs.append(borders[-1] + offset)
                    ys.append(ys[-1])
                    ax.plot(xs, ys)
            else:
                xs = ['bias'] + list(map(str, borders))
                ys = feature_explanation.expected_bias[dim] + values
                ax.bar(xs, ys)
        figs.append(fig)
    return figs