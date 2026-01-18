def json_point(t):
    g = SphereGeometry(radius=t['geometry']['size'])
    m = sage_handlers['texture'](t['texture'])
    myobject = Mesh(geometry=g, material=m, scale=[0.02, 0.02, 0.02])
    return ScaledObject(children=[myobject], position=list(t['geometry']['position']))