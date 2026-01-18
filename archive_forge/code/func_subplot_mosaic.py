from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False, width_ratios=None, height_ratios=None, empty_sentinel='.', subplot_kw=None, per_subplot_kw=None, gridspec_kw=None):
    """
        Build a layout of Axes based on ASCII art or nested lists.

        This is a helper function to build complex GridSpec layouts visually.

        See :ref:`mosaic`
        for an example and full API documentation

        Parameters
        ----------
        mosaic : list of list of {hashable or nested} or str

            A visual layout of how you want your Axes to be arranged
            labeled as strings.  For example ::

               x = [['A panel', 'A panel', 'edge'],
                    ['C panel', '.',       'edge']]

            produces 4 Axes:

            - 'A panel' which is 1 row high and spans the first two columns
            - 'edge' which is 2 rows high and is on the right edge
            - 'C panel' which in 1 row and 1 column wide in the bottom left
            - a blank space 1 row and 1 column wide in the bottom center

            Any of the entries in the layout can be a list of lists
            of the same form to create nested layouts.

            If input is a str, then it can either be a multi-line string of
            the form ::

              '''
              AAE
              C.E
              '''

            where each character is a column and each line is a row. Or it
            can be a single-line string where rows are separated by ``;``::

              'AB;CC'

            The string notation allows only single character Axes labels and
            does not support nesting but is very terse.

            The Axes identifiers may be `str` or a non-iterable hashable
            object (e.g. `tuple` s may not be used).

        sharex, sharey : bool, default: False
            If True, the x-axis (*sharex*) or y-axis (*sharey*) will be shared
            among all subplots.  In that case, tick label visibility and axis
            units behave as for `subplots`.  If False, each subplot's x- or
            y-axis will be independent.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.  Equivalent
            to ``gridspec_kw={'width_ratios': [...]}``. In the case of nested
            layouts, this argument applies only to the outer layout.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height. Equivalent
            to ``gridspec_kw={'height_ratios': [...]}``. In the case of nested
            layouts, this argument applies only to the outer layout.

        subplot_kw : dict, optional
            Dictionary with keywords passed to the `.Figure.add_subplot` call
            used to create each subplot.  These values may be overridden by
            values in *per_subplot_kw*.

        per_subplot_kw : dict, optional
            A dictionary mapping the Axes identifiers or tuples of identifiers
            to a dictionary of keyword arguments to be passed to the
            `.Figure.add_subplot` call used to create each subplot.  The values
            in these dictionaries have precedence over the values in
            *subplot_kw*.

            If *mosaic* is a string, and thus all keys are single characters,
            it is possible to use a single string instead of a tuple as keys;
            i.e. ``"AB"`` is equivalent to ``("A", "B")``.

            .. versionadded:: 3.7

        gridspec_kw : dict, optional
            Dictionary with keywords passed to the `.GridSpec` constructor used
            to create the grid the subplots are placed on. In the case of
            nested layouts, this argument applies only to the outer layout.
            For more complex layouts, users should use `.Figure.subfigures`
            to create the nesting.

        empty_sentinel : object, optional
            Entry in the layout to mean "leave this space empty".  Defaults
            to ``'.'``. Note, if *layout* is a string, it is processed via
            `inspect.cleandoc` to remove leading white space, which may
            interfere with using white-space as the empty sentinel.

        Returns
        -------
        dict[label, Axes]
           A dictionary mapping the labels to the Axes objects.  The order of
           the axes is left-to-right and top-to-bottom of their position in the
           total layout.

        """
    subplot_kw = subplot_kw or {}
    gridspec_kw = dict(gridspec_kw or {})
    per_subplot_kw = per_subplot_kw or {}
    if height_ratios is not None:
        if 'height_ratios' in gridspec_kw:
            raise ValueError("'height_ratios' must not be defined both as parameter and as key in 'gridspec_kw'")
        gridspec_kw['height_ratios'] = height_ratios
    if width_ratios is not None:
        if 'width_ratios' in gridspec_kw:
            raise ValueError("'width_ratios' must not be defined both as parameter and as key in 'gridspec_kw'")
        gridspec_kw['width_ratios'] = width_ratios
    if isinstance(mosaic, str):
        mosaic = self._normalize_grid_string(mosaic)
        per_subplot_kw = {tuple(k): v for k, v in per_subplot_kw.items()}
    per_subplot_kw = self._norm_per_subplot_kw(per_subplot_kw)
    _api.check_isinstance(bool, sharex=sharex, sharey=sharey)

    def _make_array(inp):
        """
            Convert input into 2D array

            We need to have this internal function rather than
            ``np.asarray(..., dtype=object)`` so that a list of lists
            of lists does not get converted to an array of dimension > 2.

            Returns
            -------
            2D object array
            """
        r0, *rest = inp
        if isinstance(r0, str):
            raise ValueError('List mosaic specification must be 2D')
        for j, r in enumerate(rest, start=1):
            if isinstance(r, str):
                raise ValueError('List mosaic specification must be 2D')
            if len(r0) != len(r):
                raise ValueError(f'All of the rows must be the same length, however the first row ({r0!r}) has length {len(r0)} and row {j} ({r!r}) has length {len(r)}.')
        out = np.zeros((len(inp), len(r0)), dtype=object)
        for j, r in enumerate(inp):
            for k, v in enumerate(r):
                out[j, k] = v
        return out

    def _identify_keys_and_nested(mosaic):
        """
            Given a 2D object array, identify unique IDs and nested mosaics

            Parameters
            ----------
            mosaic : 2D object array

            Returns
            -------
            unique_ids : tuple
                The unique non-sub mosaic entries in this mosaic
            nested : dict[tuple[int, int], 2D object array]
            """
        unique_ids = cbook._OrderedSet()
        nested = {}
        for j, row in enumerate(mosaic):
            for k, v in enumerate(row):
                if v == empty_sentinel:
                    continue
                elif not cbook.is_scalar_or_string(v):
                    nested[j, k] = _make_array(v)
                else:
                    unique_ids.add(v)
        return (tuple(unique_ids), nested)

    def _do_layout(gs, mosaic, unique_ids, nested):
        """
            Recursively do the mosaic.

            Parameters
            ----------
            gs : GridSpec
            mosaic : 2D object array
                The input converted to a 2D array for this level.
            unique_ids : tuple
                The identified scalar labels at this level of nesting.
            nested : dict[tuple[int, int]], 2D object array
                The identified nested mosaics, if any.

            Returns
            -------
            dict[label, Axes]
                A flat dict of all of the Axes created.
            """
        output = dict()
        this_level = dict()
        for name in unique_ids:
            indx = np.argwhere(mosaic == name)
            start_row, start_col = np.min(indx, axis=0)
            end_row, end_col = np.max(indx, axis=0) + 1
            slc = (slice(start_row, end_row), slice(start_col, end_col))
            if (mosaic[slc] != name).any():
                raise ValueError(f'While trying to layout\n{mosaic!r}\nwe found that the label {name!r} specifies a non-rectangular or non-contiguous area.')
            this_level[start_row, start_col] = (name, slc, 'axes')
        for (j, k), nested_mosaic in nested.items():
            this_level[j, k] = (None, nested_mosaic, 'nested')
        for key in sorted(this_level):
            name, arg, method = this_level[key]
            if method == 'axes':
                slc = arg
                if name in output:
                    raise ValueError(f'There are duplicate keys {name} in the layout\n{mosaic!r}')
                ax = self.add_subplot(gs[slc], **{'label': str(name), **subplot_kw, **per_subplot_kw.get(name, {})})
                output[name] = ax
            elif method == 'nested':
                nested_mosaic = arg
                j, k = key
                rows, cols = nested_mosaic.shape
                nested_output = _do_layout(gs[j, k].subgridspec(rows, cols), nested_mosaic, *_identify_keys_and_nested(nested_mosaic))
                overlap = set(output) & set(nested_output)
                if overlap:
                    raise ValueError(f'There are duplicate keys {overlap} between the outer layout\n{mosaic!r}\nand the nested layout\n{nested_mosaic}')
                output.update(nested_output)
            else:
                raise RuntimeError('This should never happen')
        return output
    mosaic = _make_array(mosaic)
    rows, cols = mosaic.shape
    gs = self.add_gridspec(rows, cols, **gridspec_kw)
    ret = _do_layout(gs, mosaic, *_identify_keys_and_nested(mosaic))
    ax0 = next(iter(ret.values()))
    for ax in ret.values():
        if sharex:
            ax.sharex(ax0)
            ax._label_outer_xaxis(skip_non_rectangular_axes=True)
        if sharey:
            ax.sharey(ax0)
            ax._label_outer_yaxis(skip_non_rectangular_axes=True)
    if (extra := (set(per_subplot_kw) - set(ret))):
        raise ValueError(f'The keys {extra} are in *per_subplot_kw* but not in the mosaic.')
    return ret