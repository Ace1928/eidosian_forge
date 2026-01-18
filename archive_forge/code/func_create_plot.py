from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Literal
from gradio_client.documentation import document
from gradio.components.plot import AltairPlot, AltairPlotData, Plot
@staticmethod
def create_plot(value: pd.DataFrame, x: str, y: str, color: str | None=None, size: str | None=None, shape: str | None=None, title: str | None=None, tooltip: list[str] | str | None=None, x_title: str | None=None, y_title: str | None=None, x_label_angle: float | None=None, y_label_angle: float | None=None, color_legend_title: str | None=None, size_legend_title: str | None=None, shape_legend_title: str | None=None, color_legend_position: Literal['left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'none'] | None=None, size_legend_position: Literal['left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'none'] | None=None, shape_legend_position: Literal['left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'none'] | None=None, height: int | None=None, width: int | None=None, x_lim: list[int | float] | None=None, y_lim: list[int | float] | None=None, interactive: bool | None=True):
    """Helper for creating the scatter plot."""
    import altair as alt
    from pandas.api.types import is_numeric_dtype
    interactive = True if interactive is None else interactive
    encodings = {'x': alt.X(x, title=x_title or x, scale=AltairPlot.create_scale(x_lim), axis=alt.Axis(labelAngle=x_label_angle) if x_label_angle is not None else alt.Axis()), 'y': alt.Y(y, title=y_title or y, scale=AltairPlot.create_scale(y_lim), axis=alt.Axis(labelAngle=y_label_angle) if y_label_angle is not None else alt.Axis())}
    properties = {}
    if title:
        properties['title'] = title
    if height:
        properties['height'] = height
    if width:
        properties['width'] = width
    if color:
        if is_numeric_dtype(value[color]):
            domain = [value[color].min(), value[color].max()]
            range_ = [0, 1]
            type_ = 'quantitative'
        else:
            domain = value[color].unique().tolist()
            range_ = list(range(len(domain)))
            type_ = 'nominal'
        encodings['color'] = {'field': color, 'type': type_, 'legend': AltairPlot.create_legend(position=color_legend_position, title=color_legend_title or color), 'scale': {'domain': domain, 'range': range_}}
    if tooltip:
        encodings['tooltip'] = tooltip
    if size:
        encodings['size'] = {'field': size, 'type': 'quantitative' if is_numeric_dtype(value[size]) else 'nominal', 'legend': AltairPlot.create_legend(position=size_legend_position, title=size_legend_title or size)}
    if shape:
        encodings['shape'] = {'field': shape, 'type': 'quantitative' if is_numeric_dtype(value[shape]) else 'nominal', 'legend': AltairPlot.create_legend(position=shape_legend_position, title=shape_legend_title or shape)}
    chart = alt.Chart(value).mark_point(clip=True).encode(**encodings).properties(background='transparent', **properties)
    if interactive:
        chart = chart.interactive()
    return chart