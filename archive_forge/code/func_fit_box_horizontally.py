from typing import Any, Optional, Tuple, Union
@classmethod
def fit_box_horizontally(cls, top: Optional[float]=None) -> 'Fit':
    """
        Display the page designated by page , with the vertical coordinate top
        positioned at the top edge of the window and the contents of the page
        magnified just enough to fit the entire width of its bounding box
        within the window.

        A null value for top specifies that the current value of that parameter
        is to be retained unchanged.

        Args:
            top:

        Returns:
            The created fit object.
        """
    return Fit(fit_type='/FitBH', fit_args=(top,))