from typing import Any, Optional, Tuple, Union
@classmethod
def fit_vertically(cls, left: Optional[float]=None) -> 'Fit':
    return Fit(fit_type='/FitV', fit_args=(left,))