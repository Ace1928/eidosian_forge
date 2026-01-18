from plotly.basedatatypes import BaseFigure
def add_traces(self, data, rows=None, cols=None, secondary_ys=None, exclude_empty_subplots=False) -> 'Figure':
    """

        Add traces to the figure

        Parameters
        ----------
        data : list[BaseTraceType or dict]
            A list of trace specifications to be added.
            Trace specifications may be either:

              - Instances of trace classes from the plotly.graph_objs
                package (e.g plotly.graph_objs.Scatter, plotly.graph_objs.Bar)
              - Dicts where:

                  - The 'type' property specifies the trace type (e.g.
                    'scatter', 'bar', 'area', etc.). If the dict has no 'type'
                    property then 'scatter' is assumed.
                  - All remaining properties are passed to the constructor
                    of the specified trace type.

        rows : None, list[int], or int (default None)
            List of subplot row indexes (starting from 1) for the traces to be
            added. Only valid if figure was created using
            `plotly.tools.make_subplots`
            If a single integer is passed, all traces will be added to row number

        cols : None or list[int] (default None)
            List of subplot column indexes (starting from 1) for the traces
            to be added. Only valid if figure was created using
            `plotly.tools.make_subplots`
            If a single integer is passed, all traces will be added to column number


        secondary_ys: None or list[boolean] (default None)
            List of secondary_y booleans for traces to be added. See the
            docstring for `add_trace` for more info.

        exclude_empty_subplots: boolean
            If True, the trace will not be added to subplots that don't already
            have traces.

        Returns
        -------
        BaseFigure
            The Figure that add_traces was called on

        Examples
        --------

        >>> from plotly import subplots
        >>> import plotly.graph_objs as go

        Add two Scatter traces to a figure

        >>> fig = go.Figure()
        >>> fig.add_traces([go.Scatter(x=[1,2,3], y=[2,1,2]),
        ...                 go.Scatter(x=[1,2,3], y=[2,1,2])]) # doctest: +ELLIPSIS
        Figure(...)

        Add two Scatter traces to vertically stacked subplots

        >>> fig = subplots.make_subplots(rows=2)
        >>> fig.add_traces([go.Scatter(x=[1,2,3], y=[2,1,2]),
        ...                 go.Scatter(x=[1,2,3], y=[2,1,2])],
        ...                 rows=[1, 2], cols=[1, 1]) # doctest: +ELLIPSIS
        Figure(...)

        """
    return super(Figure, self).add_traces(data, rows, cols, secondary_ys, exclude_empty_subplots)