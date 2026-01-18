from typing import Any, Optional, Tuple, Union
@classmethod
def fit_box(cls) -> 'Fit':
    """
        Display the page designated by page , with its contents magnified just
        enough to fit its bounding box entirely within the window both
        horizontally and vertically.

        If the required horizontal and vertical magnification factors are
        different, use the smaller of the two, centering the bounding box
        within the window in the other dimension.
        """
    return Fit(fit_type='/FitB')