from plotly.basedatatypes import BaseFigure
def select_annotations(self, selector=None, row=None, col=None, secondary_y=None):
    """
        Select annotations from a particular subplot cell and/or annotations
        that satisfy custom selection criteria.

        Parameters
        ----------
        selector: dict, function, int, str, or None (default None)
            Dict to use as selection criteria.
            Annotations will be selected if they contain properties corresponding
            to all of the dictionary's keys, with values that exactly match
            the supplied values. If None (the default), all annotations are
            selected. If a function, it must be a function accepting a single
            argument and returning a boolean. The function will be called on
            each annotation and those for which the function returned True
            will be in the selection. If an int N, the Nth annotation matching row
            and col will be selected (N can be negative). If a string S, the selector
            is equivalent to dict(type=S).
        row, col: int or None (default None)
            Subplot row and column index of annotations to select.
            To select annotations by row and column, the Figure must have been
            created using plotly.subplots.make_subplots.  To select only those
            annotation that are in paper coordinates, set row and col to the
            string 'paper'.  If None (the default), all annotations are selected.
        secondary_y: boolean or None (default None)
            * If True, only select annotations associated with the secondary
              y-axis of the subplot.
            * If False, only select annotations associated with the primary
              y-axis of the subplot.
            * If None (the default), do not filter annotations based on secondary
              y-axis.

            To select annotations by secondary y-axis, the Figure must have been
            created using plotly.subplots.make_subplots. See the docstring
            for the specs argument to make_subplots for more info on
            creating subplots with secondary y-axes.
        Returns
        -------
        generator
            Generator that iterates through all of the annotations that satisfy
            all of the specified selection criteria
        """
    return self._select_annotations_like('annotations', selector=selector, row=row, col=col, secondary_y=secondary_y)