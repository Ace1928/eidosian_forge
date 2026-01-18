import itertools
from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_update
from torchmetrics.functional.nominal.utils import (
def pearsons_contingency_coefficient_matrix(matrix: Tensor, nan_strategy: Literal['replace', 'drop']='replace', nan_replace_value: Optional[float]=0.0) -> Tensor:
    """Compute `Pearson's Contingency Coefficient`_ statistic between a set of multiple variables.

    This can serve as a convenient tool to compute Pearson's Contingency Coefficient for analyses
    of correlation between categorical variables in your dataset.

    Args:
        matrix: A tensor of categorical (nominal) data, where:

            - rows represent a number of data points
            - columns represent a number of categorical (nominal) features

        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        Pearson's Contingency Coefficient statistic for a dataset of categorical variables

    Example:
        >>> from torchmetrics.functional.nominal import pearsons_contingency_coefficient_matrix
        >>> _ = torch.manual_seed(42)
        >>> matrix = torch.randint(0, 4, (200, 5))
        >>> pearsons_contingency_coefficient_matrix(matrix)
        tensor([[1.0000, 0.2326, 0.1959, 0.2262, 0.2989],
                [0.2326, 1.0000, 0.1386, 0.1895, 0.1329],
                [0.1959, 0.1386, 1.0000, 0.1840, 0.2335],
                [0.2262, 0.1895, 0.1840, 1.0000, 0.2737],
                [0.2989, 0.1329, 0.2335, 0.2737, 1.0000]])

    """
    _nominal_input_validation(nan_strategy, nan_replace_value)
    num_variables = matrix.shape[1]
    pearsons_cont_coef_matrix_value = torch.ones(num_variables, num_variables, device=matrix.device)
    for i, j in itertools.combinations(range(num_variables), 2):
        x, y = (matrix[:, i], matrix[:, j])
        num_classes = len(torch.cat([x, y]).unique())
        confmat = _pearsons_contingency_coefficient_update(x, y, num_classes, nan_strategy, nan_replace_value)
        val = _pearsons_contingency_coefficient_compute(confmat)
        pearsons_cont_coef_matrix_value[i, j] = pearsons_cont_coef_matrix_value[j, i] = val
    return pearsons_cont_coef_matrix_value