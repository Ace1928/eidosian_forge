import warnings
def make_progress_view(self):
    """Construct a new ProgressView object for this UI.

        Application code should normally not call this but instead
        nested_progress_bar().
        """
    return NullProgressView()