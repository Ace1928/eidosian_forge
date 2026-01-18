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
def createGround(self):
    shape = BulletBoxShape(Vec3(20, 20, 1))
    body = BulletRigidBodyNode('Ground')
    body.addShape(shape)
    nodePath = self.render.attachNewNode(body)
    nodePath.set_pos(0, 0, -2)
    nodePath.set_color(0.3, 0.3, 0.3, 1)
    self.world.attachRigidBody(body)
    logging.debug('GameEnvironmentInitializer: Ground element created and positioned.')