def axis_spanning_shape_annotation(annotation, shape_type, shape_args, kwargs):
    """
    annotation: a go.layout.Annotation object, a dict describing an annotation, or None
    shape_type: one of 'vline', 'hline', 'vrect', 'hrect' and determines how the
                x, y, xanchor, and yanchor values are set.
    shape_args: the parameters used to draw the shape, which are used to place the annotation
    kwargs:     a dictionary that was the kwargs of a
                _process_multiple_axis_spanning_shapes spanning shapes call. Items in this
                dict whose keys start with 'annotation_' will be extracted and the keys with
                the 'annotation_' part stripped off will be used to assign properties of the
                new annotation.

    Property precedence:
    The annotation's x, y, xanchor, and yanchor properties are set based on the
    shape_type argument. Each property already specified in the annotation or
    through kwargs will be left as is (not replaced by the value computed using
    shape_type). Note that the xref and yref properties will in general get
    overwritten if the result of this function is passed to an add_annotation
    called with the row and col parameters specified.

    Returns an annotation populated with fields based on the
    annotation_position, annotation_ prefixed kwargs or the original annotation
    passed in to this function.
    """
    prefix = 'annotation_'
    len_prefix = len(prefix)
    annotation_keys = list(filter(lambda k: k.startswith(prefix), kwargs.keys()))
    if annotation is None and len(annotation_keys) == 0:
        return None
    if annotation is None:
        annotation = dict()
    for k in annotation_keys:
        if k == 'annotation_position':
            continue
        subk = k[len_prefix:]
        annotation[subk] = kwargs[k]
    annotation_position = None
    if 'annotation_position' in kwargs.keys():
        annotation_position = kwargs['annotation_position']
    if shape_type.endswith('line'):
        shape_dict = annotation_params_for_line(shape_type, shape_args, annotation_position)
    elif shape_type.endswith('rect'):
        shape_dict = annotation_params_for_rect(shape_type, shape_args, annotation_position)
    for k in shape_dict.keys():
        if k not in annotation or annotation[k] is None:
            annotation[k] = shape_dict[k]
    return annotation