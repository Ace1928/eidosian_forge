def json_text(t):
    tree_geometry = t['geometry']
    tree_texture = t['texture']
    tt = TextTexture(string=tree_geometry['string'])
    sm = SpriteMaterial(map=tt, opacity=tree_texture['opacity'], transparent=tree_texture['opacity'] < 1)
    return Sprite(material=sm, scaleToTexture=True)