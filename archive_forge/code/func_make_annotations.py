import plotly.colors as clrs
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
from plotly.validators.heatmap import ColorscaleValidator
def make_annotations(self):
    """
        Get annotations for each cell of the heatmap with graph_objs.Annotation

        :rtype (list[dict]) annotations: list of annotations for each cell of
            the heatmap
        """
    min_text_color, max_text_color = _AnnotatedHeatmap.get_text_color(self)
    annotations = []
    for n, row in enumerate(self.z):
        for m, val in enumerate(row):
            font_color = min_text_color if val < self.zmid else max_text_color
            annotations.append(graph_objs.layout.Annotation(text=str(self.annotation_text[n][m]), x=self.x[m], y=self.y[n], xref='x1', yref='y1', font=dict(color=font_color), showarrow=False))
    return annotations