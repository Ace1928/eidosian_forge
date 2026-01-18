from collections.abc import Iterable, Sequence
import wandb
from wandb import util
from wandb.sdk.lib import deprecate
def test_fitted(model):
    np = util.get_module('numpy', required='Logging plots requires numpy')
    pd = util.get_module('pandas', required='Logging dataframes requires pandas')
    scipy = util.get_module('scipy', required='Logging scipy matrices requires scipy')
    scikit_utils = util.get_module('sklearn.utils', required='roc requires the scikit utils submodule, install with `pip install scikit-learn`')
    scikit_exceptions = util.get_module('sklearn.exceptions', 'roc requires the scikit preprocessing submodule, install with `pip install scikit-learn`')
    try:
        model.predict(np.zeros((7, 3)))
    except scikit_exceptions.NotFittedError:
        wandb.termerror('Please fit the model before passing it in.')
        return False
    except AttributeError:
        try:
            scikit_utils.validation.check_is_fitted(model, ['coef_', 'estimator_', 'labels_', 'n_clusters_', 'children_', 'components_', 'n_components_', 'n_iter_', 'n_batch_iter_', 'explained_variance_', 'singular_values_', 'mean_'], all_or_any=any)
            return True
        except scikit_exceptions.NotFittedError:
            wandb.termerror('Please fit the model before passing it in.')
            return False
    except Exception:
        return True