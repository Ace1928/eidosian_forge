def create_from_plot(plot):
    tree = plot.scenetree_json()
    obj = sage_handlers[tree['type']](tree)
    cam = PerspectiveCamera(position=[10, 10, 10], fov=40, up=[0, 0, 1], children=[DirectionalLight(color=16777215, position=[3, 5, 1], intensity=0.5)])
    scene = Scene(children=[obj, AmbientLight(color=7829367)])
    renderer = Renderer(camera=cam, scene=scene, controls=[OrbitControls(controlling=cam)], color='white')
    return renderer