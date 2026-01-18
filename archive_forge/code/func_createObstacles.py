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
def createObstacles(self):
    for _ in range(20):
        x, y, z = (random.uniform(-18, 18), random.uniform(-18, 18), 0)
        shape = BulletBoxShape(Vec3(1, 1, 1))
        body = BulletRigidBodyNode('Obstacle')
        body.addShape(shape)
        nodePath = self.render.attachNewNode(body)
        nodePath.set_pos(x, y, z)
        self.world.attachRigidBody(body)
    logging.debug('GameEnvironmentInitializer: Obstacles created and positioned.')