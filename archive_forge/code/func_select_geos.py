from plotly.basedatatypes import BaseFigure
def select_geos(self, selector=None, row=None, col=None):
    """
        Select geo subplot objects from a particular subplot cell
        and/or geo subplot objects that satisfy custom selection
        criteria.

        Parameters
        ----------
        selector: dict, function, or None (default None)
            Dict to use as selection criteria.
            geo objects will be selected if they contain
            properties corresponding to all of the dictionary's keys, with
            values that exactly match the supplied values. If None
            (the default), all geo objects are selected. If a
            function, it must be a function accepting a single argument and
            returning a boolean. The function will be called on each
            geo and those for which the function returned True will
            be in the selection.
        row, col: int or None (default None)
            Subplot row and column index of geo objects to select.
            To select geo objects by row and column, the Figure
            must have been created using plotly.subplots.make_subplots.
            If None (the default), all geo objects are selected.
        Returns
        -------
        generator
            Generator that iterates through all of the geo
            objects that satisfy all of the specified selection criteria
        """
    return self._select_layout_subplots_by_prefix('geo', selector, row, col)