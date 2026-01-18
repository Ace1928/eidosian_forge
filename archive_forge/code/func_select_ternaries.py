from plotly.basedatatypes import BaseFigure
def select_ternaries(self, selector=None, row=None, col=None):
    """
        Select ternary subplot objects from a particular subplot cell
        and/or ternary subplot objects that satisfy custom selection
        criteria.

        Parameters
        ----------
        selector: dict, function, or None (default None)
            Dict to use as selection criteria.
            ternary objects will be selected if they contain
            properties corresponding to all of the dictionary's keys, with
            values that exactly match the supplied values. If None
            (the default), all ternary objects are selected. If a
            function, it must be a function accepting a single argument and
            returning a boolean. The function will be called on each
            ternary and those for which the function returned True will
            be in the selection.
        row, col: int or None (default None)
            Subplot row and column index of ternary objects to select.
            To select ternary objects by row and column, the Figure
            must have been created using plotly.subplots.make_subplots.
            If None (the default), all ternary objects are selected.
        Returns
        -------
        generator
            Generator that iterates through all of the ternary
            objects that satisfy all of the specified selection criteria
        """
    return self._select_layout_subplots_by_prefix('ternary', selector, row, col)