import json
import warnings
import os
from plotly import exceptions, optional_imports
from plotly.files import PLOTLY_DIR
def return_figure_from_figure_or_data(figure_or_data, validate_figure):
    from plotly.graph_objs import Figure
    from plotly.basedatatypes import BaseFigure
    validated = False
    if isinstance(figure_or_data, dict):
        figure = figure_or_data
    elif isinstance(figure_or_data, list):
        figure = {'data': figure_or_data}
    elif isinstance(figure_or_data, BaseFigure):
        figure = figure_or_data.to_dict()
        validated = True
    else:
        raise exceptions.PlotlyError('The `figure_or_data` positional argument must be `dict`-like, `list`-like, or an instance of plotly.graph_objs.Figure')
    if validate_figure and (not validated):
        try:
            figure = Figure(**figure).to_dict()
        except exceptions.PlotlyError as err:
            raise exceptions.PlotlyError("Invalid 'figure_or_data' argument. Plotly will not be able to properly parse the resulting JSON. If you want to send this 'figure_or_data' to Plotly anyway (not recommended), you can set 'validate=False' as a plot option.\nHere's why you're seeing this error:\n\n{0}".format(err))
        if not figure['data']:
            raise exceptions.PlotlyEmptyDataError("Empty data list found. Make sure that you populated the list of data objects you're sending and try again.\nQuestions? Visit support.plot.ly")
    return figure