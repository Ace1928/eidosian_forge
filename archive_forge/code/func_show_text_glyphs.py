from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def show_text_glyphs(self, text, glyphs, clusters, cluster_flags=0):
    """This operation has rendering effects similar to :meth:`show_glyphs`
        but, if the target surface supports it
        (see :meth:`Surface.has_show_text_glyphs`),
        uses the provided text and cluster mapping
        to embed the text for the glyphs shown in the output.
        If the target does not support the extended attributes,
        this method acts like the basic :meth:`show_glyphs`
        as if it had been passed ``glyphs``.

        The mapping between ``text`` and ``glyphs``
        is provided by an list of clusters.
        Each cluster covers a number of UTF-8 text bytes and glyphs,
        and neighboring clusters cover neighboring areas
        of ``text`` and ``glyphs``.
        The clusters should collectively cover ``text`` and ``glyphs``
        in entirety.

        :param text:
            The text to show, as an Unicode or UTF-8 string.
            Because of how ``cluster`` work,
            using UTF-8 bytes might be more convenient.
        :param glyphs:
            A list of glyphs.
            Each glyph is a ``(glyph_id, x, y)`` tuple.
            ``glyph_id`` is an opaque integer.
            Its exact interpretation depends on the font technology being used.
            ``x`` and ``y`` are the float offsets
            in the X and Y direction
            between the origin used for drawing or measuring the string
            and the origin of this glyph.
            Note that the offsets are not cumulative.
            When drawing or measuring text,
            each glyph is individually positioned
            with respect to the overall origin.
        :param clusters:
            A list of clusters.
            A text cluster is a minimal mapping of some glyphs
            corresponding to some UTF-8 text,
            represented as a ``(num_bytes, num_glyphs)`` tuple of integers,
            the number of UTF-8 bytes and glyphs covered by the cluster.
            For a cluster to be valid,
            both ``num_bytes`` and ``num_glyphs`` should be non-negative,
            and at least one should be non-zero.
            Note that clusters with zero glyphs
            are not as well supported as normal clusters.
            For example, PDF rendering applications
            typically ignore those clusters when PDF text is being selected.
        :type cluster_flags: int
        :param cluster_flags:
            Flags (as a bit field) for the cluster mapping.
            The first cluster always covers bytes
            from the beginning of ``text``.
            If ``cluster_flags`` does not have
            the :obj:`TEXT_CLUSTER_FLAG_BACKWARD` flag set,
            the first cluster also covers the beginning of ``glyphs``,
            otherwise it covers the end of the ``glyphs`` list
            and following clusters move backward.

        """
    glyphs = ffi.new('cairo_glyph_t[]', glyphs)
    clusters = ffi.new('cairo_text_cluster_t[]', clusters)
    cairo.cairo_show_text_glyphs(self._pointer, _encode_string(text), -1, glyphs, len(glyphs), clusters, len(clusters), cluster_flags)
    self._check_status()