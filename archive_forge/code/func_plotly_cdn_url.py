import plotly
import plotly.graph_objs as go
from plotly.offline import get_plotlyjs_version
def plotly_cdn_url(cdn_ver=get_plotlyjs_version()):
    """Return a valid plotly CDN url."""
    return 'https://cdn.plot.ly/plotly-{cdn_ver}.min.js'.format(cdn_ver=cdn_ver)