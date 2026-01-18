from plotly.basedatatypes import BaseFigure
def add_hline(self, y, row='all', col='all', exclude_empty_subplots=True, annotation=None, **kwargs) -> 'Figure':
    """

        Add a horizontal line to a plot or subplot that extends infinitely in the
        x-dimension.

        Parameters
        ----------
        y: float or int
            A number representing the y coordinate of the horizontal line.
        exclude_empty_subplots: Boolean
            If True (default) do not place the shape on subplots that have no data
            plotted on them.
        row: None, int or 'all'
            Subplot row for shape indexed starting at 1. If 'all', addresses all rows in
            the specified column(s). If both row and col are None, addresses the
            first subplot if subplots exist, or the only plot. By default is "all".
        col: None, int or 'all'
            Subplot column for shape indexed starting at 1. If 'all', addresses all rows in
            the specified column(s). If both row and col are None, addresses the
            first subplot if subplots exist, or the only plot. By default is "all".
        annotation: dict or plotly.graph_objects.layout.Annotation. If dict(),
            it is interpreted as describing an annotation. The annotation is
            placed relative to the shape based on annotation_position (see
            below) unless its x or y value has been specified for the annotation
            passed here. xref and yref are always the same as for the added
            shape and cannot be overridden.
        annotation_position: a string containing optionally ["top", "bottom"]
            and ["left", "right"] specifying where the text should be anchored
            to on the line. Example positions are "bottom left", "right top",
            "right", "bottom". If an annotation is added but annotation_position is
            not specified, this defaults to "top right".
        annotation_*: any parameters to go.layout.Annotation can be passed as
            keywords by prefixing them with "annotation_". For example, to specify the
            annotation text "example" you can pass annotation_text="example" as a
            keyword argument.
        **kwargs:
            Any named function parameters that can be passed to 'add_shape',
            except for x0, x1, y0, y1 or type.
        """
    return super(Figure, self).add_hline(y, row, col, exclude_empty_subplots, annotation, **kwargs)