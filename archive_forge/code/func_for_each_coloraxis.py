from plotly.basedatatypes import BaseFigure
def for_each_coloraxis(self, fn, selector=None, row=None, col=None) -> 'Figure':
    """
        Apply a function to all coloraxis objects that satisfy the
        specified selection criteria

        Parameters
        ----------
        fn:
            Function that inputs a single coloraxis object.
        selector: dict, function, or None (default None)
            Dict to use as selection criteria.
            coloraxis objects will be selected if they contain
            properties corresponding to all of the dictionary's keys, with
            values that exactly match the supplied values. If None
            (the default), all coloraxis objects are selected. If a
            function, it must be a function accepting a single argument and
            returning a boolean. The function will be called on each
            coloraxis and those for which the function returned True will
            be in the selection.
        row, col: int or None (default None)
            Subplot row and column index of coloraxis objects to select.
            To select coloraxis objects by row and column, the Figure
            must have been created using plotly.subplots.make_subplots.
            If None (the default), all coloraxis objects are selected.
        Returns
        -------
        self
            Returns the Figure object that the method was called on
        """
    for obj in self.select_coloraxes(selector=selector, row=row, col=col):
        fn(obj)
    return self