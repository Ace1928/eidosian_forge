import warnings
from ...utils import is_sklearn_available, requires_backends
def pearson_and_spearman(preds, labels):
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    requires_backends(pearson_and_spearman, 'sklearn')
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {'pearson': pearson_corr, 'spearmanr': spearman_corr, 'corr': (pearson_corr + spearman_corr) / 2}