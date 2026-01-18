from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def fz_new_image_from_compressed_buffer(w, h, bpc, colorspace, xres, yres, interpolate, imagemask, decode, colorkey, buffer, mask):
    """
    Class-aware wrapper for `::fz_new_image_from_compressed_buffer()`.
    	Create an image based on
    	the data in the supplied compressed buffer.

    	w,h: Width and height of the created image.

    	bpc: Bits per component.

    	colorspace: The colorspace (determines the number of components,
    	and any color conversions required while decoding).

    	xres, yres: The X and Y resolutions respectively.

    	interpolate: 1 if interpolation should be used when decoding
    	this image, 0 otherwise.

    	imagemask: 1 if this is an imagemask (i.e. transparency bitmap
    	mask), 0 otherwise.

    	decode: NULL, or a pointer to to a decode array. The default
    	decode array is [0 1] (repeated n times, for n color components).

    	colorkey: NULL, or a pointer to a colorkey array. The default
    	colorkey array is [0 255] (repeated n times, for n color
    	components).

    	buffer: Buffer of compressed data and compression parameters.
    	Ownership of this reference is passed in.

    	mask: NULL, or another image to use as a mask for this one.
    	A new reference is taken to this image. Supplying a masked
    	image as a mask to another image is illegal!
    """
    return _mupdf.fz_new_image_from_compressed_buffer(w, h, bpc, colorspace, xres, yres, interpolate, imagemask, decode, colorkey, buffer, mask)