import textwrap
import pkgutil
import copy
import os
import json
from functools import reduce
def merge_templates(self, *args):
    """
        Merge a collection of templates into a single combined template.
        Templates are process from left to right so if multiple templates
        specify the same propery, the right-most template will take
        precedence.

        Parameters
        ----------
        args: list of Template
            Zero or more template objects (or dicts with compatible properties)

        Returns
        -------
        template:
            A combined template object

        Examples
        --------

        >>> pio.templates.merge_templates(
        ...     go.layout.Template(layout={'font': {'size': 20}}),
        ...     go.layout.Template(data={'scatter': [{'mode': 'markers'}]}),
        ...     go.layout.Template(layout={'font': {'family': 'Courier'}}))
        layout.Template({
            'data': {'scatter': [{'mode': 'markers', 'type': 'scatter'}]},
            'layout': {'font': {'family': 'Courier', 'size': 20}}
        })
        """
    if args:
        return reduce(self._merge_2_templates, args)
    else:
        from plotly.graph_objs.layout import Template
        return Template()