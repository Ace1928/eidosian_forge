from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, AmbientLight, Vec4, Vec3, DirectionalLight
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerPusher, CollisionSphere
from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.bullet import BulletSphereShape
from panda3d.core import MouseWatcher, ModifierButtons, PGMouseWatcherBackground
from panda3d.core import WindowProperties
import random
import logging
def configureWindowCameraAndBackground(self):
    properties = WindowProperties()
    properties.setBackgroundColor(0.1, 0.1, 0.1)
    self.win.requestProperties(properties)
    self.disableMouse()
    self.camera.reparent_to(self.playerNodePath)
    self.camera.set_pos(0, -10, 3)
    self.camera.look_at(self.playerNodePath)
    logging.info('GameEnvironmentInitializer: Window, camera, and background color configured.')