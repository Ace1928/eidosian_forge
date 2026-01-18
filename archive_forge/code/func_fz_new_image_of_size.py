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
def fz_new_image_of_size(w, h, bpc, colorspace, xres, yres, interpolate, imagemask, decode, colorkey, mask, size, get_pixmap, get_size, drop):
    """
    Class-aware wrapper for `::fz_new_image_of_size()`.
    	Internal function to make a new fz_image structure
    	for a derived class.

    	w,h: Width and height of the created image.

    	bpc: Bits per component.

    	colorspace: The colorspace (determines the number of components,
    	and any color conversions required while decoding).

    	xres, yres: The X and Y resolutions respectively.

    	interpolate: 1 if interpolation should be used when decoding
    	this image, 0 otherwise.

    	imagemask: 1 if this is an imagemask (i.e. transparent), 0
    	otherwise.

    	decode: NULL, or a pointer to to a decode array. The default
    	decode array is [0 1] (repeated n times, for n color components).

    	colorkey: NULL, or a pointer to a colorkey array. The default
    	colorkey array is [0 255] (repeated n times, for n color
    	components).

    	mask: NULL, or another image to use as a mask for this one.
    	A new reference is taken to this image. Supplying a masked
    	image as a mask to another image is illegal!

    	size: The size of the required allocated structure (the size of
    	the derived structure).

    	get: The function to be called to obtain a decoded pixmap.

    	get_size: The function to be called to return the storage size
    	used by this image.

    	drop: The function to be called to dispose of this image once
    	the last reference is dropped.

    	Returns a pointer to an allocated structure of the required size,
    	with the first sizeof(fz_image) bytes initialised as appropriate
    	given the supplied parameters, and the other bytes set to zero.
    """
    return _mupdf.fz_new_image_of_size(w, h, bpc, colorspace, xres, yres, interpolate, imagemask, decode, colorkey, mask, size, get_pixmap, get_size, drop)