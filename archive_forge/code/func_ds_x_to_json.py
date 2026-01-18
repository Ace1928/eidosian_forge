def ds_x_to_json(ds, widget):
    return ds2json(ds, widget.zonal_speed, widget.meridional_speed, widget.latitude_dimension, widget.longitude_dimension, widget.units)