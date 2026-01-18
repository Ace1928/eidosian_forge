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
def configureDirectionalLight(self):
    directionalLight = DirectionalLight('directional_light')
    directionalLight.setColor(Vec4(0.8, 0.8, 0.8, 1))
    directionalLightNode = self.render.attachNewNode(directionalLight)
    directionalLightNode.setHpr(0, -60, 0)
    self.render.setLight(directionalLightNode)
    logging.info('GameEnvironmentInitializer: Directional light configured.')