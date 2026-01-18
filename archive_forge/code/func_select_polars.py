from plotly.basedatatypes import BaseFigure
def select_polars(self, selector=None, row=None, col=None):
    """
        Select polar subplot objects from a particular subplot cell
        and/or polar subplot objects that satisfy custom selection
        criteria.

        Parameters
        ----------
        selector: dict, function, or None (default None)
            Dict to use as selection criteria.
            polar objects will be selected if they contain
            properties corresponding to all of the dictionary's keys, with
            values that exactly match the supplied values. If None
            (the default), all polar objects are selected. If a
            function, it must be a function accepting a single argument and
            returning a boolean. The function will be called on each
            polar and those for which the function returned True will
            be in the selection.
        row, col: int or None (default None)
            Subplot row and column index of polar objects to select.
            To select polar objects by row and column, the Figure
            must have been created using plotly.subplots.make_subplots.
            If None (the default), all polar objects are selected.
        Returns
        -------
        generator
            Generator that iterates through all of the polar
            objects that satisfy all of the specified selection criteria
        """
    return self._select_layout_subplots_by_prefix('polar', selector, row, col)