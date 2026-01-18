from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
def map_face2color(face, colormap, scale, vmin, vmax):
    """
    Normalize facecolor values by vmin/vmax and return rgb-color strings

    This function takes a tuple color along with a colormap and a minimum
    (vmin) and maximum (vmax) range of possible mean distances for the
    given parametrized surface. It returns an rgb color based on the mean
    distance between vmin and vmax

    """
    if vmin >= vmax:
        raise exceptions.PlotlyError('Incorrect relation between vmin and vmax. The vmin value cannot be bigger than or equal to the value of vmax.')
    if len(colormap) == 1:
        face_color = colormap[0]
        face_color = clrs.convert_to_RGB_255(face_color)
        face_color = clrs.label_rgb(face_color)
        return face_color
    if face == vmax:
        face_color = colormap[-1]
        face_color = clrs.convert_to_RGB_255(face_color)
        face_color = clrs.label_rgb(face_color)
        return face_color
    else:
        if scale is None:
            t = (face - vmin) / float(vmax - vmin)
            low_color_index = int(t / (1.0 / (len(colormap) - 1)))
            face_color = clrs.find_intermediate_color(colormap[low_color_index], colormap[low_color_index + 1], t * (len(colormap) - 1) - low_color_index)
            face_color = clrs.convert_to_RGB_255(face_color)
            face_color = clrs.label_rgb(face_color)
        else:
            t = (face - vmin) / float(vmax - vmin)
            low_color_index = 0
            for k in range(len(scale) - 1):
                if scale[k] <= t < scale[k + 1]:
                    break
                low_color_index += 1
            low_scale_val = scale[low_color_index]
            high_scale_val = scale[low_color_index + 1]
            face_color = clrs.find_intermediate_color(colormap[low_color_index], colormap[low_color_index + 1], (t - low_scale_val) / (high_scale_val - low_scale_val))
            face_color = clrs.convert_to_RGB_255(face_color)
            face_color = clrs.label_rgb(face_color)
        return face_color