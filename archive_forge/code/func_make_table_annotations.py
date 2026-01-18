from plotly import exceptions, optional_imports
from plotly.graph_objs import graph_objs
def make_table_annotations(self):
    """
        Generate annotations to fill in table text

        :rtype (list) annotations: list of annotations for each cell of the
            table.
        """
    table_matrix = _Table.get_table_matrix(self)
    all_font_colors = _Table.get_table_font_color(self)
    annotations = []
    for n, row in enumerate(self.table_text):
        for m, val in enumerate(row):
            format_text = '<b>' + str(val) + '</b>' if n == 0 or (self.index and m < 1) else str(val)
            font_color = self.font_colors[0] if self.index and m == 0 else all_font_colors[n]
            annotations.append(graph_objs.layout.Annotation(text=format_text, x=self.x[m] - self.annotation_offset, y=self.y[n], xref='x1', yref='y1', align='left', xanchor='left', font=dict(color=font_color), showarrow=False))
    return annotations