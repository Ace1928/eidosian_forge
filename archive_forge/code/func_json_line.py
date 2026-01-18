def json_line(t):
    tree_geometry = t['geometry']
    m = sage_handlers['texture'](t['texture'])
    mesh = []
    path = [list(p) for p in tree_geometry['points']]
    mesh.append(Mesh(material=m, geometry=TubeGeometry(path=path, radialSegments=50, radius=0.01 * tree_geometry['thickness'])))
    c = Mesh(material=m, geometry=CircleGeometry(segments=50, radius=0.01 * tree_geometry['thickness']))
    c.look_at(path[0], path[1])
    c.position = path[0]
    mesh.append(c)
    if tree_geometry['arrowhead']:
        height = 0.03 * tree_geometry['thickness']
        c = Mesh(material=m, geometry=CylinderGeometry(radiusTop=0, radiusBottom=0.02 * tree_geometry['thickness'], height=height, up=[1, 0, 0], radialSegments=50))
        c.look_at(path[-1], path[-2])
        q1 = c.quaternion
        q2 = [0.7071067811865475, 0.0, 0.0, 0.7071067811865476]
        c.quaternion = [q2[3] * q1[0] + q2[0] * q1[3] - q2[1] * q1[2] + q2[2] * q1[1], q2[3] * q1[1] + q2[0] * q1[2] + q2[1] * q1[3] - q2[2] * q1[0], q2[3] * q1[2] - q2[0] * q1[1] + q2[1] * q1[0] + q2[2] * q1[3], q2[3] * q1[3] - q2[0] * q1[0] - q2[1] * q1[1] - q2[2] * q1[2]]
        p1 = path[-1]
        p2 = path[-2]
        d = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]
        last_seg = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        adjust_cone = last_seg / (height / 2)
        d = [i / adjust_cone for i in d]
        c.position = [p1[0] - d[0], p1[1] - d[1], p1[2] - d[2]]
        d2 = [i * 2 for i in d]
        if last_seg > adjust_cone * 2:
            mesh[0].geometry.path[-1] = [p1[0] - d2[0], p1[1] - d2[1], p1[2] - d2[2]]
        else:
            mesh[0].geometry.path.pop()
    else:
        c = Mesh(material=m, geometry=CircleGeometry(segments=50, radius=0.01 * tree_geometry['thickness']))
        c.look_at(path[-1], path[-2])
        c.position = path[-1]
    mesh.append(c)
    return Object3d(children=mesh)