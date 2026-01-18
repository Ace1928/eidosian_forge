import base64
import sys
from typing import Iterable, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from mlflow.exceptions import MlflowException
from mlflow.protos import facet_feature_statistics_pb2
from mlflow.recipes.cards import histogram_generator
def construct_facets_html(proto: facet_feature_statistics_pb2.DatasetFeatureStatisticsList, compare: bool=False) -> str:
    """
    Constructs the facets HTML to visualize the serialized FeatureStatisticsList proto.

    Args:
        proto: A DatasetFeatureStatisticsList proto which contains the statistics for a DataFrame.
        compare: If True, then the returned visualization switches on the comparison
            mode for several stats.

    Returns:
        The HTML for Facets visualization.
    """
    protostr = base64.b64encode(proto.SerializeToString()).decode('utf-8')
    polyfills_code = get_facets_polyfills()
    return f'\n        <div style="background-color: white">\n        <script>{polyfills_code}</script>\n        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>\n        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html" >\n        <facets-overview id="facets" proto-input="{protostr}" compare-mode="{compare}"></facets-overview>\n        </div>\n    '