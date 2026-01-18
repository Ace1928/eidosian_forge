import ctypes
import io
import operator
import os
import sys
import weakref
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontOptions, _encode_string
def dsc_comment(self, comment):
    """ Emit a comment into the PostScript output for the given surface.

        The comment is expected to conform to
        the PostScript Language Document Structuring Conventions (DSC).
        Please see that manual for details on the available comments
        and their meanings.
        In particular, the ``%%IncludeFeature`` comment allows
        a device-independent means of controlling printer device features.
        So the PostScript Printer Description Files Specification
        will also be a useful reference.

        The comment string must begin with a percent character (%)
        and the total length of the string
        (including any initial percent characters)
        must not exceed 255 bytes.
        Violating either of these conditions will
        place surface into an error state.
        But beyond these two conditions,
        this method will not enforce conformance of the comment
        with any particular specification.

        The comment string should not have a trailing newline.

        The DSC specifies different sections
        in which particular comments can appear.
        This method provides for comments to be emitted
        within three sections:
        the header, the Setup section, and the PageSetup section.
        Comments appearing in the first two sections
        apply to the entire document
        while comments in the BeginPageSetup section
        apply only to a single page.

        For comments to appear in the header section,
        this method should be called after the surface is created,
        but before a call to :meth:`dsc_begin_setup`.

        For comments to appear in the Setup section,
        this method should be called after a call to :meth:`dsc_begin_setup`
        but before a call to :meth:`dsc_begin_page_setup`.

        For comments to appear in the PageSetup section,
        this method should be called after a call to
        :meth:`dsc_begin_page_setup`.

        Note that it is only necessary to call :meth:`dsc_begin_page_setup`
        for the first page of any surface.
        After a call to :meth:`~Surface.show_page`
        or :meth:`~Surface.copy_page`
        comments are unambiguously directed
        to the PageSetup section of the current page.
        But it doesn't hurt to call this method
        at the beginning of every page
        as that consistency may make the calling code simpler.

        As a final note,
        cairo automatically generates several comments on its own.
        As such, applications must not manually generate
        any of the following comments:

        Header section: ``%!PS-Adobe-3.0``, ``%%Creator``, ``%%CreationDate``,
        ``%%Pages``, ``%%BoundingBox``, ``%%DocumentData``,
        ``%%LanguageLevel``, ``%%EndComments``.

        Setup section: ``%%BeginSetup``, ``%%EndSetup``.

        PageSetup section: ``%%BeginPageSetup``, ``%%PageBoundingBox``,
        ``%%EndPageSetup``.

        Other sections: ``%%BeginProlog``, ``%%EndProlog``, ``%%Page``,
        ``%%Trailer``, ``%%EOF``.

        """
    cairo.cairo_ps_surface_dsc_comment(self._pointer, _encode_string(comment))
    self._check_status()